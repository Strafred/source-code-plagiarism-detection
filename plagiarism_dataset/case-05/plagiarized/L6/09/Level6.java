import java.util.Scanner;
public class Level6 {
    public static void main(String[] args) {
        System.out.print("Enter an integer: ");
        Scanner input = new Scanner(System.in);
        int bil = input.nextInt();
        rekursif(bil);
        System.out.println("");
    }
    public static int rekursif(int bil){
        if(bil!=0){
            int sisaMod = bil % 10;
            System.out.print(sisaMod);
            bil = bil / 10;
            rekursif(bil);
        }
        return bil;
    }
}
