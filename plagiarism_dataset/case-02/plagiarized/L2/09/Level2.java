import java.util.Scanner;
public class Level2 {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        System.out.print("Enter the radius and length of a cylinder: ");
        double rad = in.nextDouble();
        double panjang = in.nextDouble();
        double luas = rad * rad * 3.14159;
        double vol = luas * panjang;
        System.out.println("The area is " + luas);
        System.out.println("The volume of the cylinder is " + vol);
    }
}
