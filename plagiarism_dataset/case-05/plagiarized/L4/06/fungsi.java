import java.util.Scanner;
public class fungsi {
    public static void main(String[] args) {
         System.out.print("Enter an integer: ");
        Scanner sc = new Scanner(System.in);
        int   input;
        input =    sc.nextInt();
        while (input != 0) {
               int sisa ;
               sisa= input % 10;
               System.out.print(sisa);
               input = input / 10;
        }
    }
}
