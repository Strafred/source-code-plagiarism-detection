import java.util.Scanner;
public class T05 {
    public static void main(String[] args) {
          Scanner s = new Scanner(System.in);
          System.out.print("Enter an integer ");
          int a= s.nextInt();
          reverse(a);
    }
    public  static  void reverse(int number){
        String cek = String.valueOf(number);
        for (int i = 0; i < cek.length(); i++) {
            System.out.print(cek.charAt(cek.length()-1-i));
        }
        System.out.println("");
    }
}
